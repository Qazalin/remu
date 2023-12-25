
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n31>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v2, 0                                        // 000000001610: 7E040280
	s_lshl_b64 s[6:7], s[4:5], 2                               // 000000001614: 84868204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b32 s8, s[2:3], null                                // 000000001624: F4000201 F8000000
	s_mov_b32 s3, 0                                            // 00000000162C: BE830080
	s_waitcnt lgkmcnt(0)                                       // 000000001630: BF89FC07
	s_bfe_u32 s6, s8, 0x80017                                  // 000000001634: 9306FF08 00080017
	s_and_b32 s2, s8, 0x7fffff                                 // 00000000163C: 8B02FF08 007FFFFF
	s_sub_i32 s7, 0x96, s6                                     // 000000001644: 818706FF 00000096
	s_bitset1_b32 s2, 23                                       // 00000000164C: BE821297
	s_add_i32 s9, s6, 0xffffff6a                               // 000000001650: 8109FF06 FFFFFF6A
	s_add_i32 s10, s6, 0xffffff81                              // 000000001658: 810AFF06 FFFFFF81
	s_lshr_b64 s[6:7], s[2:3], s7                              // 000000001660: 85860702
	s_lshl_b64 s[2:3], s[2:3], s9                              // 000000001664: 84820902
	s_cmp_gt_i32 s10, 23                                       // 000000001668: BF02970A
	s_cselect_b32 s3, s3, s7                                   // 00000000166C: 98030703
	s_cselect_b32 s2, s2, s6                                   // 000000001670: 98020602
	s_ashr_i32 s6, s8, 31                                      // 000000001674: 86069F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001678: BF870499
	s_ashr_i32 s7, s6, 31                                      // 00000000167C: 86079F06
	s_xor_b64 s[2:3], s[2:3], s[6:7]                           // 000000001680: 8D820602
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001684: BF870009
	s_sub_u32 s2, s2, s6                                       // 000000001688: 80820602
	s_subb_u32 s3, s3, s7                                      // 00000000168C: 82830703
	s_cmp_lt_i32 s10, 0                                        // 000000001690: BF04800A
	s_cselect_b32 s6, 0, s3                                    // 000000001694: 98060380
	s_cselect_b32 s7, 0, s2                                    // 000000001698: 98070280
	s_lshl_b64 s[2:3], s[4:5], 3                               // 00000000169C: 84828304
	v_dual_mov_b32 v0, s7 :: v_dual_mov_b32 v1, s6             // 0000000016A0: CA100007 00000006
	s_add_u32 s0, s0, s2                                       // 0000000016A8: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000016AC: 82010301
	global_store_b64 v2, v[0:1], s[0:1]                        // 0000000016B0: DC6E0000 00000002
	s_nop 0                                                    // 0000000016B8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016BC: BFB60003
	s_endpgm                                                   // 0000000016C0: BFB00000
