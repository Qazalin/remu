
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n42>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[6:7], s[4:5], 3                               // 000000001614: 84868304
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b64 s[2:3], s[2:3], null                            // 000000001624: F4040081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	s_clz_i32_u32 s6, s3                                       // 000000001630: BE860A03
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001634: BF870499
	s_min_u32 s6, s6, 32                                       // 000000001638: 8986A006
	s_lshl_b64 s[2:3], s[2:3], s6                              // 00000000163C: 84820602
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001640: BF870499
	s_min_u32 s2, s2, 1                                        // 000000001644: 89828102
	s_or_b32 s2, s3, s2                                        // 000000001648: 8C020203
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000164C: BF870009
	v_cvt_f32_u32_e32 v0, s2                                   // 000000001650: 7E000C02
	s_sub_i32 s2, 32, s6                                       // 000000001654: 818206A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001658: BF870481
	v_ldexp_f32 v0, v0, s2                                     // 00000000165C: D71C0000 00000500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001664: BF870091
	v_cmp_class_f32_e64 s2, v0, 0x90                           // 000000001668: D47E0002 0001FF00 00000090
	v_cndmask_b32_e64 v1, 1.0, 0x4f800000, s2                  // 000000001674: D5010001 0009FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001680: BF8704B1
	v_mul_f32_e32 v0, v1, v0                                   // 000000001684: 10000101
	v_cndmask_b32_e64 v1, 0, 0x42000000, s2                    // 000000001688: D5010001 0009FE80 42000000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001694: 84828204
	s_add_u32 s0, s0, s2                                       // 000000001698: 80000200
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000169C: BF8700C2
	v_log_f32_e32 v0, v0                                       // 0000000016A0: 7E004F00
	s_addc_u32 s1, s1, s3                                      // 0000000016A4: 82010301
	s_waitcnt_depctr 0xfff                                     // 0000000016A8: BF880FFF
	v_dual_sub_f32 v0, v0, v1 :: v_dual_mov_b32 v1, 0          // 0000000016AC: C9500300 00000080
	v_mul_f32_e32 v0, 0x3f317218, v0                           // 0000000016B4: 100000FF 3F317218
	global_store_b32 v1, v0, s[0:1]                            // 0000000016BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000016C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016C8: BFB60003
	s_endpgm                                                   // 0000000016CC: BFB00000
