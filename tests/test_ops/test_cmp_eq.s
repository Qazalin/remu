
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_60>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001618: 86039F0F
	v_mov_b32_e32 v1, 0                                        // 00000000161C: 7E020280
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001620: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001624: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001628: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000162C: 82070307
	s_add_u32 s0, s0, s2                                       // 000000001630: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001634: 82010301
	s_load_b32 s6, s[6:7], null                                // 000000001638: F4000183 F8000000
	s_load_b32 s0, s[0:1], null                                // 000000001640: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001648: BF89FC07
	v_cmp_lt_f32_e64 s1, s0, s6                                // 00000000164C: D4110001 00000C00
	v_cmp_lt_f32_e64 vcc_lo, s6, s0                            // 000000001654: D411006A 00000006
	s_add_u32 s0, s4, s2                                       // 00000000165C: 80000204
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001660: BF8700A2
	v_cndmask_b32_e64 v0, 0, 1, s1                             // 000000001664: D5010000 00050280
	s_addc_u32 s1, s5, s3                                      // 00000000166C: 82010305
	v_add_co_ci_u32_e32 v0, vcc_lo, 0, v0, vcc_lo              // 000000001670: 40000080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001674: BF870091
	v_cvt_f32_ubyte0_e32 v0, v0                                // 000000001678: 7E002300
	v_sub_f32_e32 v0, 1.0, v0                                  // 00000000167C: 080000F2
	global_store_b32 v1, v0, s[0:1]                            // 000000001680: DC6A0000 00000001
	s_nop 0                                                    // 000000001688: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000168C: BFB60003
	s_endpgm                                                   // 000000001690: BFB00000
