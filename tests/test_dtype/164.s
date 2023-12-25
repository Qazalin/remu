
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n65>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	v_mov_b32_e32 v2, 0                                        // 000000001614: 7E040280
	s_ashr_i32 s3, s15, 31                                     // 000000001618: 86039F0F
	s_mov_b32 s2, s15                                          // 00000000161C: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 000000001620: BF89FC07
	s_add_u32 s6, s6, s15                                      // 000000001624: 80060F06
	s_addc_u32 s7, s7, s3                                      // 000000001628: 82070307
	s_lshl_b64 s[2:3], s[2:3], 3                               // 00000000162C: 84828302
	global_load_i8 v0, v2, s[6:7]                              // 000000001630: DC460000 00060002
	s_add_u32 s0, s0, s2                                       // 000000001638: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000163C: 82010301
	s_load_b64 s[0:1], s[0:1], null                            // 000000001640: F4040000 F8000000
	s_waitcnt vmcnt(0)                                         // 000000001648: BF8903F7
	v_bfe_i32 v0, v0, 0, 16                                    // 00000000164C: D6110000 02410100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001654: BF870141
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001658: 3402009F
	s_waitcnt lgkmcnt(0)                                       // 00000000165C: BF89FC07
	v_add_co_u32 v0, vcc_lo, s0, v0                            // 000000001660: D7006A00 00020000
	s_add_u32 s0, s4, s2                                       // 000000001668: 80000204
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo             // 00000000166C: 40020201
	s_addc_u32 s1, s5, s3                                      // 000000001670: 82010305
	global_store_b64 v2, v[0:1], s[0:1]                        // 000000001674: DC6E0000 00000002
	s_nop 0                                                    // 00000000167C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001680: BFB60003
	s_endpgm                                                   // 000000001684: BFB00000
