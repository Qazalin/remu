
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n23>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_mov_b32 s4, s15                                          // 000000001610: BE84000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001618: 80020F02
	s_addc_u32 s3, s3, s5                                      // 00000000161C: 82030503
	global_load_u8 v1, v0, s[2:3]                              // 000000001620: DC420000 01020000
	s_waitcnt vmcnt(0)                                         // 000000001628: BF8903F7
	v_cvt_f32_ubyte0_e32 v1, v1                                // 00000000162C: 7E022301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001630: BF870091
	v_cmp_class_f32_e64 s2, v1, 0x90                           // 000000001634: D47E0002 0001FF01 00000090
	v_cndmask_b32_e64 v2, 1.0, 0x4f800000, s2                  // 000000001640: D5010002 0009FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000164C: BF8704B1
	v_mul_f32_e32 v1, v2, v1                                   // 000000001650: 10020302
	v_cndmask_b32_e64 v2, 0, 0x42000000, s2                    // 000000001654: D5010002 0009FE80 42000000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001660: 84828204
	s_add_u32 s0, s0, s2                                       // 000000001664: 80000200
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001668: BF8700C2
	v_log_f32_e32 v1, v1                                       // 00000000166C: 7E024F01
	s_addc_u32 s1, s1, s3                                      // 000000001670: 82010301
	s_waitcnt_depctr 0xfff                                     // 000000001674: BF880FFF
	v_sub_f32_e32 v1, v1, v2                                   // 000000001678: 08020501
	v_mul_f32_e32 v1, 0x3f317218, v1                           // 00000000167C: 100202FF 3F317218
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001684: BF870001
	v_mul_f32_e32 v1, 0x3fb8aa3b, v1                           // 000000001688: 100202FF 3FB8AA3B
	global_store_b32 v0, v1, s[0:1]                            // 000000001690: DC6A0000 00000100
	s_nop 0                                                    // 000000001698: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000169C: BFB60003
	s_endpgm                                                   // 0000000016A0: BFB00000
