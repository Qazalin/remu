
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n40>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v1, 0                                        // 000000001610: 7E020280
	s_lshl_b64 s[6:7], s[4:5], 3                               // 000000001614: 84868304
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b64 s[2:3], s[2:3], null                            // 000000001624: F4040081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	s_xor_b32 s6, s2, s3                                       // 000000001630: 8D060302
	s_cls_i32 s7, s3                                           // 000000001634: BE870C03
	s_ashr_i32 s6, s6, 31                                      // 000000001638: 86069F06
	s_add_i32 s7, s7, -1                                       // 00000000163C: 8107C107
	s_add_i32 s6, s6, 32                                       // 000000001640: 8106A006
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001644: BF870499
	s_min_u32 s6, s7, s6                                       // 000000001648: 89860607
	s_lshl_b64 s[2:3], s[2:3], s6                              // 00000000164C: 84820602
	s_sub_i32 s6, 32, s6                                       // 000000001650: 818606A0
	s_min_u32 s2, s2, 1                                        // 000000001654: 89828102
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001658: BF870499
	s_or_b32 s2, s3, s2                                        // 00000000165C: 8C020203
	v_cvt_f32_i32_e32 v0, s2                                   // 000000001660: 7E000A02
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001664: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001668: BF8700A9
	s_add_u32 s0, s0, s2                                       // 00000000166C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001670: 82010301
	v_ldexp_f32 v0, v0, s6                                     // 000000001674: D71C0000 00000D00
	global_store_b32 v1, v0, s[0:1]                            // 00000000167C: DC6A0000 00000001
	s_nop 0                                                    // 000000001684: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001688: BFB60003
	s_endpgm                                                   // 00000000168C: BFB00000
